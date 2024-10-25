import { App } from 'aws-cdk-lib';
import { POCStack } from '.';

const pocApp = new App();
const pocPackageName = pocApp.node.tryGetContext('PACKAGE_NAME');
const pocDescription: string | undefined = pocApp.node.tryGetContext('POC_DESCRIPTION');
const extensionOnly: boolean = pocApp.node.tryGetContext('EXTENSION_ONLY') || false;

new POCStack(pocApp, pocPackageName, {
  pocName: pocPackageName,
  pocPackageName: pocPackageName,
  pocDescription: pocDescription,
  extensionOnly: extensionOnly,
});